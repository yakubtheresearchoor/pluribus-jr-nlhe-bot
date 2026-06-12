//! Clean-room 6-max NLHE dealer + betting state machine + settlement.
//! The harness supplies the deck order (duplicate-play control); this
//! module deals, enforces legality, tracks streets, builds layered
//! side pots, applies the rake spec, and settles via the clean-room
//! evaluator. Money conservation (Σ net + rake = 0) is asserted on
//! every settlement — an always-on invariant, not just a test.
//!
//! Rake spec (matches the project's stated rule, implemented fresh):
//! rake = clamp(main-pot total × rate, 0, cap), taken from the MAIN
//! pot only, only when the flop was seen (no-flop-no-drop).

use crate::eval::best5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
    Showdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Fold,
    Check,
    Call,
    /// Raise TO this total street commitment (a bet is a raise from 0).
    RaiseTo(u32),
}

#[derive(Clone, Debug)]
pub struct TableConfig {
    pub num_players: usize,
    pub sb: u32,
    pub bb: u32,
    pub stacks: Vec<u32>,
    /// Rake in milli-units of the pot (e.g. 50 = 5%).
    pub rake_milli: u32,
    pub rake_cap: u32,
}

#[derive(Clone, Debug)]
pub struct Settlement {
    /// Chip delta per seat for the whole hand (winnings − all money
    /// put in). Σ net = −rake.
    pub net: Vec<i64>,
    pub rake: u32,
    /// Showdown verdicts the harness cross-checks against the solver:
    /// (seat, best-5 rank) for every player who reached showdown.
    pub showdown_ranks: Vec<(usize, u32)>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RulesError {
    NotPlayersTurn,
    IllegalAction(&'static str),
    HandOver,
}

#[derive(Clone, Debug)]
pub struct HandState {
    cfg: TableConfig,
    deck: Vec<u8>,
    pub button: usize,
    pub street: Street,
    pub board: Vec<u8>,
    pub holes: Vec<[u8; 2]>,
    pub folded: Vec<bool>,
    pub all_in: Vec<bool>,
    /// Total committed this street.
    pub street_commit: Vec<u32>,
    /// Total committed this hand.
    pub total_commit: Vec<u32>,
    pub stacks: Vec<u32>,
    /// Seat to act (None when the hand or street needs advancing).
    pub to_act: Option<usize>,
    /// Highest street commitment to match.
    pub current_bet: u32,
    /// Size of the last raise increment (min-raise rule).
    pub last_raise: u32,
    /// Players still owed an action this street.
    pending: Vec<bool>,
    flop_seen: bool,
}

impl HandState {
    /// Deal a hand: deck is the FULL 52-card order (harness-controlled;
    /// duplicate play replays the same order with rotated seats).
    /// Dealing order: two hole cards to each seat starting left of the
    /// button (one at a time, as dealt live), then board streets with
    /// burns.
    pub fn new(cfg: TableConfig, button: usize, deck: Vec<u8>) -> Self {
        assert_eq!(deck.len(), 52);
        {
            let mut seen = [false; 52];
            for &c in &deck {
                assert!(!seen[c as usize], "duplicate card in deck");
                seen[c as usize] = true;
            }
        }
        let np = cfg.num_players;
        assert!((2..=9).contains(&np));
        let mut holes = vec![[0u8; 2]; np];
        let mut di = 0;
        for round in 0..2 {
            for k in 1..=np {
                let seat = (button + k) % np;
                holes[seat][round] = deck[di];
                di += 1;
            }
        }
        let mut stacks = cfg.stacks.clone();
        assert_eq!(stacks.len(), np);
        let sb_seat = if np == 2 { button } else { (button + 1) % np };
        let bb_seat = (sb_seat + 1) % np;
        let mut street_commit = vec![0u32; np];
        let mut total_commit = vec![0u32; np];
        let mut all_in = vec![false; np];
        let mut post = |seat: usize, amt: u32| {
            let a = amt.min(stacks[seat]);
            stacks[seat] -= a;
            street_commit[seat] += a;
            total_commit[seat] += a;
            if stacks[seat] == 0 {
                all_in[seat] = true;
            }
        };
        post(sb_seat, cfg.sb);
        post(bb_seat, cfg.bb);
        let first = (bb_seat + 1) % np;
        let mut s = HandState {
            current_bet: cfg.bb,
            last_raise: cfg.bb,
            cfg,
            deck,
            button,
            street: Street::Preflop,
            board: Vec::new(),
            holes,
            folded: vec![false; np],
            all_in,
            street_commit,
            total_commit,
            stacks,
            to_act: Some(first),
            pending: vec![true; np],
            flop_seen: false,
        };
        s.skip_unable();
        s.maybe_advance();
        s
    }

    fn np(&self) -> usize {
        self.cfg.num_players
    }
    fn live(&self, p: usize) -> bool {
        !self.folded[p] && !self.all_in[p]
    }
    fn alive_count(&self) -> usize {
        (0..self.np()).filter(|&p| !self.folded[p]).count()
    }

    /// Advance to_act past folded / all-in seats.
    fn skip_unable(&mut self) {
        for _ in 0..self.np() {
            match self.to_act {
                Some(p) if !self.live(p) || !self.pending[p] => {
                    self.to_act = Some((p + 1) % self.np());
                }
                _ => return,
            }
        }
        self.to_act = None; // nobody can act
    }

    fn street_done(&self) -> bool {
        (0..self.np()).all(|p| {
            !self.live(p) || (!self.pending[p] && self.street_commit[p] == self.current_bet)
                || (!self.pending[p] && self.street_commit[p] < self.current_bet && self.all_in[p])
        })
    }

    /// Deal the next street(s); runs the board out when nobody can act.
    fn maybe_advance(&mut self) {
        loop {
            if self.alive_count() <= 1 {
                self.street = Street::Showdown;
                self.to_act = None;
                return;
            }
            if !self.street_done() {
                self.skip_unable();
                if self.to_act.is_some() {
                    return;
                }
            }
            // Street complete — deal next.
            let np = self.np();
            let dealt = 2 * np;
            match self.street {
                Street::Preflop => {
                    // burn + 3
                    let b = dealt + 1;
                    self.board.extend_from_slice(&self.deck[b..b + 3]);
                    self.street = Street::Flop;
                    self.flop_seen = true;
                }
                Street::Flop => {
                    let b = dealt + 5;
                    self.board.push(self.deck[b]);
                    self.street = Street::Turn;
                }
                Street::Turn => {
                    let b = dealt + 7;
                    self.board.push(self.deck[b]);
                    self.street = Street::River;
                }
                Street::River => {
                    self.street = Street::Showdown;
                    self.to_act = None;
                    return;
                }
                Street::Showdown => return,
            }
            for p in 0..np {
                self.street_commit[p] = 0;
                self.pending[p] = true;
            }
            self.current_bet = 0;
            self.last_raise = self.cfg.bb;
            self.to_act = Some((self.button + 1) % np);
            self.skip_unable();
            if self.to_act.is_some() {
                return;
            }
            // Everyone folded/all-in: loop to deal the next street.
        }
    }

    pub fn is_over(&self) -> bool {
        self.street == Street::Showdown
    }

    /// Legal actions for the seat to act (RaiseTo carries min/max).
    pub fn legal(&self) -> Option<(usize, Vec<Action>, u32, u32)> {
        let p = self.to_act?;
        let owe = self.current_bet - self.street_commit[p];
        let mut acts = Vec::new();
        if owe > 0 {
            acts.push(Action::Fold);
            acts.push(Action::Call);
        } else {
            acts.push(Action::Check);
        }
        let max_to = self.street_commit[p] + self.stacks[p];
        let min_to = (self.current_bet + self.last_raise).min(max_to);
        if max_to > self.current_bet {
            acts.push(Action::RaiseTo(min_to));
        }
        Some((p, acts, min_to, max_to))
    }

    pub fn apply(&mut self, seat: usize, a: Action) -> Result<(), RulesError> {
        let Some(p) = self.to_act else { return Err(RulesError::HandOver) };
        if p != seat {
            return Err(RulesError::NotPlayersTurn);
        }
        let owe = self.current_bet - self.street_commit[p];
        match a {
            Action::Fold => {
                if owe == 0 {
                    return Err(RulesError::IllegalAction("fold facing no bet"));
                }
                self.folded[p] = true;
            }
            Action::Check => {
                if owe != 0 {
                    return Err(RulesError::IllegalAction("check facing a bet"));
                }
            }
            Action::Call => {
                if owe == 0 {
                    return Err(RulesError::IllegalAction("call facing no bet"));
                }
                let pay = owe.min(self.stacks[p]);
                self.stacks[p] -= pay;
                self.street_commit[p] += pay;
                self.total_commit[p] += pay;
                if self.stacks[p] == 0 {
                    self.all_in[p] = true;
                }
            }
            Action::RaiseTo(to) => {
                let max_to = self.street_commit[p] + self.stacks[p];
                if to <= self.current_bet {
                    return Err(RulesError::IllegalAction("raise not above current bet"));
                }
                if to > max_to {
                    return Err(RulesError::IllegalAction("raise beyond stack"));
                }
                let full_min = self.current_bet + self.last_raise;
                if to < full_min && to != max_to {
                    return Err(RulesError::IllegalAction("short raise (not all-in)"));
                }
                let inc = to - self.current_bet;
                let pay = to - self.street_commit[p];
                self.stacks[p] -= pay;
                self.street_commit[p] = to;
                self.total_commit[p] += pay;
                if self.stacks[p] == 0 {
                    self.all_in[p] = true;
                }
                if inc >= self.last_raise {
                    self.last_raise = inc;
                    // A full raise reopens action for everyone else.
                    for q in 0..self.np() {
                        if q != p {
                            self.pending[q] = true;
                        }
                    }
                }
                self.current_bet = to;
            }
        }
        self.pending[p] = false;
        self.to_act = Some((p + 1) % self.np());
        self.maybe_advance();
        Ok(())
    }

    /// Settle the finished hand: layered side pots, rake from the main
    /// pot only (no-flop-no-drop), ties split with odd chips to the
    /// earliest eligible seat left of the button (deterministic rule).
    pub fn settle(&self) -> Settlement {
        assert!(self.is_over(), "settle before hand over");
        let np = self.np();

        // Showdown ranks for live players (the cross-check surface).
        let live: Vec<usize> = (0..np).filter(|&p| !self.folded[p]).collect();
        let mut showdown_ranks = Vec::new();
        let ranks: Vec<Option<u32>> = (0..np)
            .map(|p| {
                if self.folded[p] {
                    None
                } else if live.len() > 1 {
                    let mut cards = self.holes[p].to_vec();
                    cards.extend_from_slice(&self.board);
                    let r = best5(&cards).0;
                    showdown_ranks.push((p, r));
                    Some(r)
                } else {
                    Some(0) // uncontested — rank irrelevant
                }
            })
            .collect();

        let rake_spec = if self.flop_seen {
            (self.cfg.rake_milli, self.cfg.rake_cap)
        } else {
            (0, 0)
        };
        let net = settle_pots(&self.total_commit, &self.folded, &ranks, self.button, rake_spec);
        Settlement { net, rake: rake_of(&self.total_commit, &self.folded, rake_spec), showdown_ranks }

    }
}


/// Rake actually taken under (milli, cap) on the main pot (after the
/// uncalled refund). Pass (0, 0) when no flop was seen.
pub fn rake_of(total_commit: &[u32], folded: &[bool], rake_spec: (u32, u32)) -> u32 {
    let np = total_commit.len();
    let mut commit: Vec<u32> = total_commit.to_vec();
    let mut idx: Vec<usize> = (0..np).collect();
    idx.sort_unstable_by_key(|&p| std::cmp::Reverse(commit[p]));
    if commit[idx[0]] > commit[idx[1]] {
        commit[idx[0]] = commit[idx[1]];
    }
    let _ = folded;
    let mut levels: Vec<u32> = commit.iter().copied().filter(|&c| c > 0).collect();
    levels.sort_unstable();
    levels.dedup();
    let total_pot: u32 = commit.iter().sum();
    let main_level = levels.first().copied().unwrap_or(0);
    let main_pot: u32 =
        commit.iter().map(|&c| c.min(main_level)).sum::<u32>().min(total_pot);
    ((main_pot as u64 * rake_spec.0 as u64 / 1000) as u32).min(rake_spec.1)
}

/// THE pot-settlement function: clean-room layered side pots with
/// uncalled refund, rake on the main pot, ties split (odd chips to
/// the earliest eligible seat left of the button). `ranks[p]` is the
/// showdown strength of live players (any consistent value when a
/// player wins uncontested). Returns net chip delta per seat;
/// CONSERVATION (Σ net = −rake) is asserted — an always-on invariant
/// the harness inherits for every settled hand.
pub fn settle_pots(
    total_commit: &[u32],
    folded: &[bool],
    ranks: &[Option<u32>],
    button: usize,
    rake_spec: (u32, u32),
) -> Vec<i64> {
    let np = total_commit.len();
    let mut net: Vec<i64> = (0..np).map(|p| -(total_commit[p] as i64)).collect();
    let rake = rake_of(total_commit, folded, rake_spec);
        // Uncalled-bet refund (live rule): the strict-largest total
        // commitment refunds down to the second-largest before pots
        // form. (A folded player can never hold the strict max: you
        // only fold facing a commitment at least your own.) All other
        // folded money stays in the pot.
        let mut commit: Vec<u32> = total_commit.to_vec();
        {
            let mut idx: Vec<usize> = (0..np).collect();
            idx.sort_unstable_by_key(|&p| std::cmp::Reverse(commit[p]));
            let top = idx[0];
            let second = commit[idx[1]];
            if commit[top] > second {
                let refund = commit[top] - second;
                net[top] += refund as i64;
                commit[top] = second;
            }
        }

        // Layered pots from sorted distinct commitment levels.
        let mut levels: Vec<u32> = commit.iter().copied().filter(|&c| c > 0).collect();
        levels.sort_unstable();
        levels.dedup();
        let mut rake_left = rake;

        let mut prev = 0u32;
        for &lev in &levels {
            let mut pot: u64 = 0;
            for p in 0..np {
                pot += (commit[p].min(lev).saturating_sub(prev)) as u64;
            }
            // Rake comes off the first (main) pot layer(s).
            let r = (rake_left as u64).min(pot);
            pot -= r;
            rake_left -= r as u32;
            // Eligible: not folded and committed to this level. After
            // the refund, every layer has at least one live claimant.
            let elig: Vec<usize> = (0..np)
                .filter(|&p| !folded[p] && commit[p] >= lev)
                .collect();
            assert!(!elig.is_empty(), "pot layer with no eligible live player");
            let best = elig.iter().map(|&p| ranks[p].unwrap()).max().unwrap();
            let winners: Vec<usize> =
                elig.iter().copied().filter(|&p| ranks[p].unwrap() == best).collect();
            let share = pot / winners.len() as u64;
            let mut odd = pot - share * winners.len() as u64;
            // Odd chips: earliest winner left of the button.
            let mut order: Vec<usize> = winners.clone();
            order.sort_by_key(|&p| (p + np - (button + 1) % np) % np);
            for w in order {
                let extra = if odd > 0 { 1 } else { 0 };
                odd -= extra;
                net[w] += (share + extra) as i64;
            }
            prev = lev;
        }

        // ALWAYS-ON INVARIANT: money conserves (Σ net = −rake).
        let total: i64 = net.iter().sum();
        assert_eq!(
            total,
            -(rake as i64),
            "money conservation violated: Σnet {total} rake {rake}"
        );

    let total: i64 = net.iter().sum();
    assert_eq!(total, -(rake as i64), "money conservation violated");
    net
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(rank: u32, suit: u32) -> u8 {
        (rank * 4 + suit) as u8
    }

    /// Deck builder: forced cards at dealing positions, rest filled
    /// with the remaining cards in order.
    fn deck_with(forced: &[(usize, u8)]) -> Vec<u8> {
        let mut deck = vec![255u8; 52];
        let mut used = [false; 52];
        for &(i, c) in forced {
            deck[i] = c;
            used[c as usize] = true;
        }
        let mut fill = (0..52u8).filter(|&c| !used[c as usize]);
        for slot in deck.iter_mut() {
            if *slot == 255 {
                *slot = fill.next().unwrap();
            }
        }
        deck
    }

    fn cfg3(stacks: [u32; 3]) -> TableConfig {
        TableConfig {
            num_players: 3,
            sb: 1,
            bb: 2,
            stacks: stacks.to_vec(),
            rake_milli: 0,
            rake_cap: 0,
        }
    }

    /// HAND-BUILT side-pot scenario (3-way all-in, layered pots
    /// computed by hand in the comments):
    ///   button=0, sb=1 (stack 10), bb=2 (stack 20), btn stack 30.
    ///   P0 raises to 30 (all-in), P1 calls (10 all-in), P2 calls (20
    ///   all-in). Commits: [30, 10, 20]. Uncalled refund: 30→20 (P0
    ///   gets 10 back). Main pot 10×3=30 → P1 (AA). Side pot
    ///   (10..20]: 10×2=20 → P2 (KK beats 72o).
    ///   Nets: P0 −20, P1 +20, P2 0.
    #[test]
    fn three_way_allin_side_pots_by_hand() {
        // Dealing: round0 → seats 1,2,0 = deck[0,1,2]; round1 → deck[3,4,5].
        let deck = deck_with(&[
            (0, card(12, 2)), // P1: Ah
            (3, card(12, 3)), // P1: As
            (1, card(11, 0)), // P2: Kc
            (4, card(11, 1)), // P2: Kd
            (2, card(5, 0)),  // P0: 7c
            (5, card(0, 1)),  // P0: 2d
            // Board: blanks (3c 8d 9h Jc / 4d at burns' positions).
            (7, card(1, 0)),
            (8, card(6, 1)),
            (9, card(7, 2)),
            (11, card(9, 0)),
            (13, card(2, 1)),
        ]);
        let mut h = HandState::new(cfg3([30, 10, 20]), 0, deck);
        // Preflop order after blinds: P0 (utg=button at 3-handed).
        h.apply(0, Action::RaiseTo(30)).unwrap();
        h.apply(1, Action::Call).unwrap();
        h.apply(2, Action::Call).unwrap();
        assert!(h.is_over());
        let s = h.settle();
        assert_eq!(s.rake, 0);
        assert_eq!(s.net, vec![-20, 20, 0]);
        // All three reached showdown.
        assert_eq!(s.showdown_ranks.len(), 3);
    }

    /// Uncontested pot: everyone folds preflop; winner takes blinds,
    /// no rake (no flop), exact by hand.
    #[test]
    fn fold_around_exact_blinds() {
        let mut h = HandState::new(cfg3([100, 100, 100]), 0, deck_with(&[]));
        h.apply(0, Action::Fold).unwrap();
        h.apply(1, Action::Fold).unwrap(); // sb folds; bb wins
        assert!(h.is_over());
        let s = h.settle();
        assert_eq!(s.rake, 0, "no flop, no drop");
        assert_eq!(s.net, vec![0, -1, 1]);
        assert!(s.showdown_ranks.is_empty());
    }

    /// Rake: heads-up showdown, pot 40, 5% rake cap 3 → rake 2
    /// (40×0.05), winner nets +18, loser −20.
    #[test]
    fn rake_main_pot_by_hand() {
        let cfg = TableConfig {
            num_players: 2,
            sb: 1,
            bb: 2,
            stacks: vec![100, 100],
            rake_milli: 50,
            rake_cap: 3,
        };
        // HU: button is sb and acts first preflop.
        let deck = deck_with(&[
            (0, card(12, 2)), // P1 (bb in HU? seat order: round0 seats (btn+1)%2=1 then 0)
            (2, card(12, 3)),
            (1, card(0, 0)),
            (3, card(1, 1)),
            (5, card(6, 0)),
            (6, card(7, 1)),
            (7, card(9, 2)),
            (9, card(2, 3)),
            (11, card(4, 2)),
        ]);
        let mut h = HandState::new(cfg, 0, deck);
        // P0 = button/sb acts first preflop.
        h.apply(0, Action::RaiseTo(20)).unwrap();
        h.apply(1, Action::Call).unwrap();
        // Postflop: bb (seat 1) acts first.
        h.apply(1, Action::Check).unwrap();
        h.apply(0, Action::Check).unwrap();
        h.apply(1, Action::Check).unwrap();
        h.apply(0, Action::Check).unwrap();
        h.apply(1, Action::Check).unwrap();
        h.apply(0, Action::Check).unwrap();
        assert!(h.is_over());
        let s = h.settle();
        assert_eq!(s.rake, 2);
        let total: i64 = s.net.iter().sum();
        assert_eq!(total, -2);
        assert_eq!(s.net.iter().max().unwrap() + 0, 18);
    }

    /// Conservation + no-panic fuzz: seeded LCG plays random legal
    /// actions across thousands of hands; every settlement must
    /// conserve money (asserted inside settle()) and every state
    /// transition must be legal.
    #[test]
    fn conservation_fuzz() {
        let mut rng: u64 = 0x5EED_CAFE;
        let mut next = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rng >> 33
        };
        for hand in 0..5000u32 {
            let np = 2 + (next() % 5) as usize; // 2..6 players
            let stacks: Vec<u32> = (0..np).map(|_| 20 + (next() % 200) as u32).collect();
            let cfg = TableConfig {
                num_players: np,
                sb: 1,
                bb: 2,
                stacks,
                rake_milli: if hand % 2 == 0 { 50 } else { 0 },
                rake_cap: 3,
            };
            // Random deck: Fisher-Yates with the LCG.
            let mut deck: Vec<u8> = (0..52).collect();
            for i in (1..52usize).rev() {
                let j = (next() % (i as u64 + 1)) as usize;
                deck.swap(i, j);
            }
            let mut h = HandState::new(cfg, (next() % np as u64) as usize, deck);
            let mut steps = 0;
            while !h.is_over() {
                steps += 1;
                assert!(steps < 1000, "state machine did not terminate");
                let (p, acts, min_to, max_to) = h.legal().expect("someone to act");
                let pick = &acts[(next() % acts.len() as u64) as usize];
                let a = match pick {
                    Action::RaiseTo(_) => {
                        let span = (max_to - min_to) as u64 + 1;
                        Action::RaiseTo(min_to + (next() % span) as u32)
                    }
                    other => *other,
                };
                h.apply(p, a).expect("legal action rejected");
            }
            let _ = h.settle(); // conservation asserted inside
        }
    }
}
